"""Element registry.

Maps the short names a mesh uses onto element classes, so a new element can be
made available to the solvers without editing fem2d source:

    from fem2d.elements import Element, RegisterElement, ShapeDerivative

    class MyQuad(Element):
        NumNodes = 4
        QuadOrder = 2
        def ShapeFunctions(self, xi, eta): ...
        def ShapeDerivatives(self, xi, eta): ...

    class dMyQuaddX(ShapeDerivative, MyQuad):
        pass

    RegisterElement('MyQuad', MyQuad, dMyQuaddX)

A mesh may also be given the class itself instead of a registered name, in
which case the derivative class is looked up from the registry if it was
registered, and sensitivities are unavailable if it was not.
"""

_REGISTRY = {}


def RegisterElement(Name: str, ElementClass: type, DerivativeClass: type | None = None) -> type:
    '''
    Parameters
    ----------
    Name : Short name a mesh refers to the element by, for example 'Q4'.
    ElementClass : Element subclass.
    DerivativeClass : Matching shape derivative class, needed only if
                      sensitivities are wanted for this element.

    Returns
    -------
    ElementClass, so this can be used as a decorator.
    '''

    _REGISTRY[Name] = (ElementClass, DerivativeClass)

    return ElementClass


def RegisteredElements() -> list[str]:
    '''
    Returns
    -------
    Sorted list of the registered element names.
    '''

    return sorted(_REGISTRY)


def LookupElement(ElementType, Derivative: bool = False) -> type:
    '''
    Parameters
    ----------
    ElementType : Registered name, or an Element subclass directly.
    Derivative : Return the shape derivative class instead of the element.

    Returns
    -------
    The element class, or its shape derivative counterpart.
    '''

    if isinstance(ElementType, type):

        # A class was passed directly. Find its registered derivative, if any.
        Name = ElementType.__name__

        Registered = next((Pair for Pair in _REGISTRY.values() if Pair[0] is ElementType), None)

        DerivativeClass = Registered[1] if Registered is not None else None

        if not Derivative:

            return ElementType

        if DerivativeClass is None:

            raise ValueError(
                f'No shape derivative class is registered for {Name}, so sensitivities '
                f'are unavailable. Register one with RegisterElement(name, {Name}, dMyElementdX).')

        return DerivativeClass

    if ElementType not in _REGISTRY:

        raise ValueError(f'Unknown ElementType {ElementType!r}, expected one of {RegisteredElements()} or an '
                         'Element subclass.')

    Element, DerivativeClass = _REGISTRY[ElementType]

    if Derivative:

        if DerivativeClass is None:

            raise ValueError('No shape derivative class is registered for '
                             f'{ElementType!r}, so sensitivities are unavailable.')

        return DerivativeClass

    return Element
